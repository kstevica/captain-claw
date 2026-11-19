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

log = structlog.get_logger(__name__)

ENGINE = "funnel"
RECIPE = "shared/recipe.json"
STATE = "shared/funnel"
ITEMS = f"{STATE}/items.csv"      # every item looked at: what became of it, and why
ROWS = f"{STATE}/rows.csv"        # the kept rows, the merge's input
MAYBE = f"{STATE}/maybe.csv"      # nearly there: for an agent step to finish (phase 2)
QUERIES = f"{STATE}/queries.json"  # searches made, queries used, new-query rounds

SEARCH_BUDGET = 1000
FETCH_CHARS = 6000
CONCURRENCY = 4
MAX_QUERY_ROUNDS = 2
ITEMS_MAX = 5000     # items looked at per step (config funnel_items_max)
READ_CHARS = 12_000  # what one item's judging call reads of its pages, in all
MORE_PAGES = 10      # list pages found in search results, followed per step
DECIDE_TOKENS = 8192
NOTE_EVERY = 10

SOURCE_KINDS = ("search", "page", "links", "file", "document")
FILTER_OPS = ("regex", "not_regex", "in", "not_in", "min", "max", "size_max", "size_min")
CHECK_OPS = ("required", "in", "regex", "in_evidence", "within_months", "min", "max")
_ITEM_COLS = ["key", "group", "status", "reason", "source", "step", "at"]
_GROUP_COL = "_group"
_SKIP_SITES = ("linkedin.com", "facebook.com", "twitter.com", "x.com", "instagram.com", "youtube.com",
               "wikipedia.org", "crunchbase.com", "zoominfo.com", "rocketreach.co", "bbb.org", "yelp.com",
               "clutch.co", "glassdoor.com", "indeed.com", "apollo.io", "dnb.com", "bloomberg.com")


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
        "item": str(raw.get("item") or "item").strip()[:60] or "item",
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
    return recipe


def _source(s: Any, groups: dict[str, int]) -> dict[str, Any] | None:
    if not isinstance(s, dict) or s.get("kind") not in SOURCE_KINDS:
        return None
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
 "decide_batch": <items per judging call: 1 when each item has pages to read, up to 20 for short rows>}

Rules:
- Use the owner's own files and links first; they cost nothing to read. Search only when the list has to be
  found on the web, and then write 10-25 searches per group that find lists of the right items (directories,
  associations, "N-M employees" pages, rankings), not one item each.
- Gather only what the checks and the rules need, and put the cheapest test first: a size band from a search
  snippet ("size_max" filter) before reading pages.
- Templates name columns in braces ({Firm}, {Website}) or {domain}.
- checks: "in_evidence" for every value copied from a source (an email, a date, a quote) with the column that
  holds its link; "within_months" for a date window; "in" for fixed values.
- An offline list (a file or documents only) has no searches and no gather unless the rules need the web.
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
            if not p.is_file() or p.name.startswith(".") or p in seen or len(lines) > 60:
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


def default_recipe(folder: Path, base: dict[str, Any]) -> dict[str, Any] | None:
    """A recipe without a model: the folder's own tables, documents and links as
    sources. None when there's nothing to read."""
    sources = []
    for place in (folder / "extra", folder):
        if not place.is_dir():
            continue
        for p in sorted(place.iterdir()):
            if not p.is_file() or p.name.startswith("."):
                continue
            rel = p.relative_to(folder).as_posix()
            if rel == base.get("output"):
                continue
            ext = p.suffix.lower()
            if ext in fx.TABLE_EXTS:
                sources.append({"kind": "file", "path": rel})
            elif ext in fx.DOC_EXTS:
                sources.append({"kind": "document", "path": rel})
    if not sources:
        return None
    return normalize_recipe({"sources": sources}, base)


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
                    resp = await prov.complete(messages=[
                        Message(role="system", content=SETUP_SYSTEM
                                + ("\nKeep your thinking short; the answer is JSON only." if attempt else "")),
                        Message(role="user", content=prompt),
                    ], temperature=0.2, max_tokens=mt)
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
        recipe = await asyncio.to_thread(default_recipe, folder, base)
        if recipe is not None:
            why = f"set up from the folder's own files ({why})" if why else ""
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
wasn't in what was read. drop: it breaks a rule. Dates as YYYY-MM-DD, or YYYY-MM when there's no day.
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
        self.more_pages: list[str] = []
        self.tools: Tools | None = None
        self.creds: dict | None = None
        self.seen: set[str] = set()
        self.kept: dict[str, int] = {}
        self.counts = {"looked": 0, "kept": 0, "maybe": 0, "dropped": 0}
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

    async def _ask(self, label: str, system: str, user: str) -> Any:
        from captain_claw.flight_deck import bat_routes
        from captain_claw.llm import Message
        creds = await self._creds()
        async with bat_routes._metered(self.run["id"], f"funnel-{label}"):
            prov, mt = bat_routes.think_call(creds, temperature=0.0, default_max=DECIDE_TOKENS)
            resp = await prov.complete(messages=[Message(role="system", content=system),
                                                 Message(role="user", content=user)],
                                       temperature=0.0, max_tokens=mt)
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
        out = []
        for it in items or []:
            if not isinstance(it, dict) or not str(it.get("name") or "").strip():
                continue
            fields = {str(k): str(v) for k, v in (it.get("fields") or {}).items() if v not in (None, "")}
            fields.setdefault("name", str(it["name"]).strip())
            if it.get("url"):
                fields.setdefault("url", str(it["url"]).strip())
            out.append({"fields": fields, "origin": origin, "text": text if len(text) <= 3000 else ""})
        return out

    def item_key(self, item: dict) -> str:
        f = {k.lower(): v for k, v in item["fields"].items()}
        key = self.recipe["key"]
        site = self.recipe.get("site_column") or ""
        v = f.get(key.lower(), "") if key and key != "domain" else ""
        v = v or f.get(site.lower(), "") or f.get("url", "") or f.get("website", "") or f.get("name", "")
        return norm_key(v)

    def passes(self, item: dict, found: dict[str, list[tuple[str, str]]] | None = None) -> str:
        """"" when an item passes the recipe's filters, else why not. Before gathering
        only the item's fields are known; a size band is tested once it's found."""
        f = {k.lower(): v for k, v in item["fields"].items()}
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
            if self.recipe.get("site_column") and any(fx.same_site(key, s) for s in _SKIP_SITES):
                continue  # items with sites of their own: a directory or a social page isn't one
            item["key"] = key
            self.seen.add(key)
            self.counts["looked"] += 1
            why = self.passes(item)
            if why:
                await self.record(item, "dropped", why)
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
                await self.record(it, "dropped", why)
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
                                  f"# Columns\n{cols}{groups}\n\n# Items\n" + "\n\n".join(lines))
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
                    await self.record(it, "dropped", f"check: {problem}")
                    continue
                group = row.get(self.recipe["group_by"], "") if self.recipe["group_by"] else ""
                group = next((g for g in self.recipe["groups"] if g.lower() == group.lower()), group) or self.group
                if self.quota(group) and self.kept.get(group, 0) >= self.quota(group):
                    await self.record(it, "dropped", f"{group} is already full")
                    continue
                key = norm_key(row.get(self.recipe["key"], "")) if self.recipe["key"] in row else it["key"]
                if key and key != it["key"] and key in self.seen:
                    await self.record(it, "dropped", "already looked at under another name")
                    continue
                self.seen.add(key)
                self.kept[group] = self.kept.get(group, 0) + 1
                _append_csv(self.folder / ROWS, self.recipe["columns"] + [_GROUP_COL], {**row, _GROUP_COL: group})
                await self.record(it, "kept", reason or "meets the rules", group=group)
            elif verdict == "maybe":
                await self.record(it, "maybe", reason or "a fact it needs wasn't found")
                _append_csv(self.folder / MAYBE, ["key", "fields", "why", "sources"],
                            {"key": it["key"], "fields": json.dumps(it["fields"], ensure_ascii=False),
                             "why": reason, "sources": " ".join(e["source"] for e in it.get("evidence") or [])})
            else:
                await self.record(it, "dropped", reason or "breaks a rule")

    def check(self, row: dict[str, str], item: dict) -> str:
        """Why a row the model kept fails the recipe's checks by code, or ""."""
        texts = {e["source"]: e["text"] for e in item.get("evidence") or []}
        every = "\n".join(texts.values())
        for c in self.recipe["checks"]:
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

    async def record(self, item: dict, status: str, reason: str, group: str = "") -> None:
        self.counts["kept" if status == "kept" else status if status == "maybe" else "dropped"] += 1
        if status == "dropped":
            short = re.sub(r"\(.*?\)|'[^']*'|\d[\d,.-]*", "", reason).strip()[:60]
            self.reasons[short] = self.reasons.get(short, 0) + 1
        name = item["fields"].get("name") or next(iter(item["fields"].values()), "") or item.get("key", "")
        _append_csv(self.folder / ITEMS, _ITEM_COLS,
                    {"key": item.get("key", ""), "group": group or self.group, "status": status, "reason": reason,
                     "source": item.get("origin", ""), "step": self.key, "at": time.strftime("%Y-%m-%d %H:%M")})
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
                      f"{c['dropped']} dropped. Searches used in the run: {self.queries['searches']} of {self.budget}."]
        if self.reasons:
            top = sorted(self.reasons.items(), key=lambda x: -x[1])[:6]
            lines += ["", "Dropped most often for:"] + [f"- {r or 'other'} ({n})" for r, n in top]
        if self.recent:
            lines += ["", "Latest:"] + [f"- {r}" for r in self.recent]
        p = self.folder / "progress" / f"{bat_routes._step_file(self.key)}.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")

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
               f"Searches used in the run: {self.queries['searches']} of {self.budget}.")
        return {"ok": True, "output": out, "usd": 0.0, "tokens": 0, "error": ""}


# ── the merge ──────────────────────────────────────────────────────────

def merge(folder: Path, recipe: dict[str, Any]) -> dict:
    """The kept rows as the list (its columns in order, no two with the same key),
    and every dropped item with why (``exclusions.csv``)."""
    rows = _read_csv(folder / ROWS)
    cols = recipe["columns"]
    out, seen = [], set()
    for r in rows:
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
    dropped = [r for r in _read_csv(folder / ITEMS) if r.get("status") == "dropped"]
    with (folder / "exclusions.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["key", "group", "reason", "source"], extrasaction="ignore")
        w.writeheader()
        w.writerows(dropped)
    per = ""
    if recipe["groups"]:
        counts = {g: 0 for g in recipe["groups"]}
        for r in rows:
            g = r.get(_GROUP_COL, "")
            counts[g] = counts.get(g, 0) + 1
        per = " " + "; ".join(f"{g} {counts.get(g, 0)}" + (f" of {n}" if n else "") for g, n in recipe["groups"].items()) + "."
    maybe = len(_read_csv(folder / MAYBE))
    text = (f"`{recipe['output']}`: {len(out)} rows" + (f" of {recipe['target']}" if recipe.get("target") else "")
            + f".{per} `exclusions.csv`: {len(dropped)} items dropped, with why."
            + (f" {maybe} nearly there in `{MAYBE}`." if maybe else ""))
    return {"ok": True, "output": text, "usd": 0.0, "tokens": 0, "error": ""}


async def attempt(run: dict, step: dict) -> dict:
    """A funnel step, in place of a Bat worker's turn."""
    from captain_claw.flight_deck import bat_routes
    kind = step_kind(run, step) or {}
    folder = bat_routes._run_folder(run)
    recipe = load_recipe(folder)
    if recipe is None:
        return {"ok": False, "error": f"the list funnel's recipe ({RECIPE}) is missing", "usd": 0.0, "tokens": 0}
    if kind.get("kind") == "merge":
        return await asyncio.to_thread(merge, folder, recipe)
    work = Work(run, step, recipe, folder, kind.get("group", ""))
    try:
        return await work.run_step()
    except RuntimeError as e:
        return {"ok": False, "error": str(e), "usd": 0.0, "tokens": 0}
    finally:
        await work.close()
